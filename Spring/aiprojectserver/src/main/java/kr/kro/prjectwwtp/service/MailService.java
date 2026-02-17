package kr.kro.prjectwwtp.service;



import java.io.ByteArrayOutputStream;
import java.nio.ByteBuffer;
import java.nio.charset.StandardCharsets;
import java.util.Base64;
import java.util.List;
import java.util.Properties;

import org.springframework.beans.factory.annotation.Value;
import org.springframework.stereotype.Service;

import com.sendgrid.Method;
import com.sendgrid.Request;
import com.sendgrid.Response;
import com.sendgrid.SendGrid;
import com.sendgrid.helpers.mail.Mail;
import com.sendgrid.helpers.mail.objects.Attachments;
import com.sendgrid.helpers.mail.objects.Content;
import com.sendgrid.helpers.mail.objects.Email;

//import com.amazonaws.services.simpleemail.AmazonSimpleEmailService;
//import com.amazonaws.services.simpleemail.model.Body;
//import com.amazonaws.services.simpleemail.model.Content;
//import com.amazonaws.services.simpleemail.model.Destination;
//import com.amazonaws.services.simpleemail.model.Message;
//import com.amazonaws.services.simpleemail.model.RawMessage;
//import com.amazonaws.services.simpleemail.model.SendEmailRequest;
//import com.amazonaws.services.simpleemail.model.SendEmailResult;
//import com.amazonaws.services.simpleemail.model.SendRawEmailRequest;
//import com.amazonaws.services.simpleemail.model.SendRawEmailResult;

//import jakarta.activation.DataHandler;
//import jakarta.mail.Session;
//import jakarta.mail.internet.InternetAddress;
//import jakarta.mail.internet.MimeBodyPart;
//import jakarta.mail.internet.MimeMessage;
//import jakarta.mail.internet.MimeMultipart;
//import jakarta.mail.util.ByteArrayDataSource;
import kr.kro.prjectwwtp.domain.Member;
import lombok.RequiredArgsConstructor;

@Service
@RequiredArgsConstructor
public class MailService {
//	private final AmazonSimpleEmailService amazonSimpleEmailService;
	private final SendGrid sendGrid;
	private final LogService logService;
//	
//	@Value("${aws.region}")
//	private String region;
//	@Value("${aws.ses.send-mail-from}")
//	private String sendMailFrom;
	
	@Value("${spring.sendgrid.api-key}")
    private String apiKey;

    @Value("${spring.sendgrid.from-email}")
    private String fromEmail;	

	public void sendEmail(Member member, String subject, String bodyHtml) {
        String type = "send One";
		String messageId = null;
		String errorMsg = null;
		try {
//			Destination destination = new Destination().withToAddresses(member.getUserEmail());
//			SendEmailResult result = sendEmail(destination, subjectText, bodyText);
//	        messageId = result.getMessageId();
//			System.out.println("Email send response: " + result);
			Email from = new Email(fromEmail);
			Email to = new Email(member.getUserEmail());
			Content content = new Content("text/html", bodyHtml);
			Mail mail = new Mail(from, subject, to, content);
			
			Request request = new Request();
			request.setMethod(Method.POST);
			request.setEndpoint("mail/send");
			request.setBody(mail.build());
			
			Response response = sendGrid.api(request);
			System.out.println("sendEmail response : " + response.getStatusCode());
			if(response.getStatusCode() != 202)
				errorMsg = response.getBody();
		} catch(Exception e) {
			errorMsg = e.getMessage();
		} finally {
			logService.addMailLog(member, type, messageId, errorMsg);	
		}
	}
	
	public void sendEmail(List<String> addressList, String subjectText, String bodyText) {
        String type = "send All";
        String messageId = null;
		String errorMsg = null;
        try {
//			Destination destination = new Destination().withToAddresses(addressList);
//			SendEmailResult result = sendEmail(destination, subjectText, bodyText);
//			messageId = result.getMessageId();
//			System.out.println("Email send response: " + result);
        } catch (Exception e) {
			errorMsg = e.getMessage();
		} finally {
			logService.addMailLog(null, type, messageId, errorMsg);	
		}
	}
	
//	private SendEmailResult sendEmail(Destination destination, String subjectText, String bodyText) {
//		Content subject = new Content().withCharset("UTF-8").withData(subjectText);
//		Content body = new Content().withCharset("UTF-8").withData(bodyText);
//		
//		Message message = new Message().withSubject(subject)
//				.withBody(new Body().withHtml(body));
//		
//		SendEmailRequest  request = new SendEmailRequest()
//			.withSource(sendMailFrom)
//			.withDestination(destination)
//			.withMessage(message);
//		
//		return amazonSimpleEmailService.sendEmail(request);
//	}
//	
//	public void sendEmailWithAttachment(Member member, String subject, String bodyHtml, String fileContent, String fileName) {
//	    // 1. 메일 세션 설정
//		Session session = Session.getDefaultInstance(new Properties());
//		String type = "sendReport";
//		String messageId = null;
//		String errorMsg = null;
//
//	    try {
//	        // 2. MIME 메시지 생성
//	    	MimeMessage message = new MimeMessage(session);
//	    	message.setSubject(subject, "UTF-8");
//	        message.setFrom(new InternetAddress("kyuhuhu.sujidaddy@gmail.com"));
//	        message.setRecipients(MimeMessage.RecipientType.TO, InternetAddress.parse(member.getUserEmail()));
//
//	        // 3. 메일의 여러 부분을 담을 Multipart 생성 (mixed: 본문 + 첨부파일)
//	        MimeMultipart multipart = new MimeMultipart("mixed");
//
//	        // -- (A) 본문 부분 추가 (HTML) --
//	        MimeBodyPart htmlPart = new MimeBodyPart();
//	        htmlPart.setContent(bodyHtml, "text/html; charset=UTF-8");
//	        multipart.addBodyPart(htmlPart);
//
//	        // -- (B) 첨부 파일 부분 추가 (.txt) --
//	        MimeBodyPart attachmentPart = new MimeBodyPart();
//	        // 텍스트 데이터를 DataSource로 변환
//	        ByteArrayDataSource dataSource = new ByteArrayDataSource(fileContent.getBytes("UTF-8"), "text/plain");
//	        attachmentPart.setDataHandler(new DataHandler(dataSource));
//	        attachmentPart.setFileName(fileName); // 파일 이름 설정
//	        multipart.addBodyPart(attachmentPart);
//
//	        // 4. 메시지에 Multipart 설정
//	        message.setContent(multipart);
//
//	        // 5. AWS SES 전송을 위한 Raw 데이터 변환
//	        ByteArrayOutputStream outputStream = new ByteArrayOutputStream();
//	        message.writeTo(outputStream);
//	        
//	        // RawMessage 생성
//	        RawMessage rawMessage = new RawMessage(ByteBuffer.wrap(outputStream.toByteArray()));
//
//	        // 6. SES 클라이언트를 통한 전송
//	        SendRawEmailRequest rawEmailRequest = new SendRawEmailRequest(rawMessage);
//	        SendRawEmailResult result = amazonSimpleEmailService.sendRawEmail(rawEmailRequest);
//	        // 추적을 위한 ID, 로그 기록에 추가해야할듯
//	        messageId = result.getMessageId();
//	        System.out.println("Email send response: " + result);
//
//	    } catch (Exception e) {
//	        errorMsg = e.getMessage();
//	    } finally {
//			logService.addMailLog(member, type, messageId, errorMsg);	
//		}
//	}
	public void sendEmailWithAttachment(Member member, String subject, String bodyHtml, String fileContent, String fileName) {
		String type = "sendReport";
        String messageId = null;
		String errorMsg = null;
        try {
        	Email from = new Email(fromEmail);
			Email to = new Email(member.getUserEmail());
			Content content = new Content("text/html", bodyHtml);
			Mail mail = new Mail(from, subject, to, content);
			
			mail.addAttachments(createAttachment(fileContent, fileName));
			
			Request request = new Request();
			request.setMethod(Method.POST);
			request.setEndpoint("mail/send");
			request.setBody(mail.build());
			
			Response response = sendGrid.api(request);
			System.out.println("sendEmail response : " + response.getStatusCode());
			if(response.getStatusCode() != 202)
				errorMsg = response.getBody();
        } catch (Exception e) {
			errorMsg = e.getMessage();
		} finally {
			logService.addMailLog(null, type, messageId, errorMsg);	
		}
	}
	
	private Attachments createAttachment(String fileContent, String fileName) {
		Attachments attachment = new Attachments();
		attachment.setContent(Base64.getEncoder().encodeToString(fileContent.getBytes(StandardCharsets.UTF_8)));
		attachment.setFilename(fileName);
		attachment.setType("text/html");
		attachment.setDisposition("attachment");
		return attachment;
	}
}
