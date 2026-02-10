package kr.kro.prjectwwtp.service;



import java.io.ByteArrayOutputStream;
import java.nio.ByteBuffer;
import java.util.List;
import java.util.Properties;

import org.springframework.beans.factory.annotation.Value;
import org.springframework.stereotype.Service;

import com.amazonaws.services.simpleemail.AmazonSimpleEmailService;
import com.amazonaws.services.simpleemail.model.Body;
import com.amazonaws.services.simpleemail.model.Content;
import com.amazonaws.services.simpleemail.model.Destination;
import com.amazonaws.services.simpleemail.model.Message;
import com.amazonaws.services.simpleemail.model.RawMessage;
import com.amazonaws.services.simpleemail.model.SendEmailRequest;
import com.amazonaws.services.simpleemail.model.SendEmailResult;
import com.amazonaws.services.simpleemail.model.SendRawEmailRequest;

import jakarta.activation.DataHandler;
import jakarta.mail.Session;
import jakarta.mail.internet.InternetAddress;
import jakarta.mail.internet.MimeBodyPart;
import jakarta.mail.internet.MimeMessage;
import jakarta.mail.internet.MimeMultipart;
import jakarta.mail.util.ByteArrayDataSource;
import lombok.RequiredArgsConstructor;

@Service
@RequiredArgsConstructor
public class MailService {
	private final AmazonSimpleEmailService amazonSimpleEmailService;
	
	@Value("${aws.region}")
	private String region;
	@Value("${aws.ses.send-mail-from}")
	private String sendMailFrom;

	public void sendEmail(String toAddress, String subjectText, String bodyText) {
		Destination destination = new Destination().withToAddresses(toAddress);
		sendEmail(destination, subjectText, bodyText);
	}
	
	public void sendEmail(List<String> addressList, String subjectText, String bodyText) {
		Destination destination = new Destination().withToAddresses(addressList);
		sendEmail(destination, subjectText, bodyText);
	}
	
	private void sendEmail(Destination destination, String subjectText, String bodyText) {
		Content subject = new Content().withCharset("UTF-8").withData(subjectText);
		Content body = new Content().withCharset("UTF-8").withData(bodyText);
		
		Message message = new Message().withSubject(subject)
				.withBody(new Body().withHtml(body));
		
		SendEmailRequest  request = new SendEmailRequest()
			.withSource(sendMailFrom)
			.withDestination(destination)
			.withMessage(message);
		
		SendEmailResult result = amazonSimpleEmailService.sendEmail(request);
		System.out.println("Email send response: " + result);
	}
	
	public void sendEmailWithAttachment(List<String> addressList, String subject, String bodyHtml, String fileContent, String fileName) {
	    // 1. 메일 세션 설정
		Session session = Session.getDefaultInstance(new Properties());

	    try {
	        // 2. MIME 메시지 생성
	    	MimeMessage message = new MimeMessage(session);
	    	message.setSubject(subject, "UTF-8");
	        message.setFrom(new InternetAddress("kyuhuhu.sujidaddy@gmail.com"));
	        String to = String.join(", ", addressList);
	        message.setRecipients(MimeMessage.RecipientType.TO, InternetAddress.parse(to));

	        // 3. 메일의 여러 부분을 담을 Multipart 생성 (mixed: 본문 + 첨부파일)
	        MimeMultipart multipart = new MimeMultipart("mixed");

	        // -- (A) 본문 부분 추가 (HTML) --
	        MimeBodyPart htmlPart = new MimeBodyPart();
	        htmlPart.setContent(bodyHtml, "text/html; charset=UTF-8");
	        multipart.addBodyPart(htmlPart);

	        // -- (B) 첨부 파일 부분 추가 (.txt) --
	        MimeBodyPart attachmentPart = new MimeBodyPart();
	        // 텍스트 데이터를 DataSource로 변환
	        ByteArrayDataSource dataSource = new ByteArrayDataSource(fileContent.getBytes("UTF-8"), "text/plain");
	        attachmentPart.setDataHandler(new DataHandler(dataSource));
	        attachmentPart.setFileName(fileName); // 파일 이름 설정
	        multipart.addBodyPart(attachmentPart);

	        // 4. 메시지에 Multipart 설정
	        message.setContent(multipart);

	        // 5. AWS SES 전송을 위한 Raw 데이터 변환
	        ByteArrayOutputStream outputStream = new ByteArrayOutputStream();
	        message.writeTo(outputStream);
	        
	        // RawMessage 생성
	        RawMessage rawMessage = new RawMessage(ByteBuffer.wrap(outputStream.toByteArray()));

	        // 6. SES 클라이언트를 통한 전송
	        SendRawEmailRequest rawEmailRequest = new SendRawEmailRequest(rawMessage);
	        amazonSimpleEmailService.sendRawEmail(rawEmailRequest);

	        System.out.println("첨부파일이 포함된 이메일이 성공적으로 전송되었습니다.");

	    } catch (Exception e) {
	        System.err.println("이메일 전송 중 오류 발생: " + e.getMessage());
	        e.printStackTrace();
	    }
	}
}
