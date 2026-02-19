package kr.kro.prjectwwtp.config;

import java.security.Key;
import java.util.Base64;

import javax.crypto.Cipher;
import javax.crypto.spec.IvParameterSpec;
import javax.crypto.spec.SecretKeySpec;

import org.springframework.beans.factory.annotation.Value;
import org.springframework.stereotype.Service;

import jakarta.persistence.AttributeConverter;
import jakarta.persistence.Converter;
import lombok.RequiredArgsConstructor;

@Service
@Converter
@RequiredArgsConstructor
public class CryptoStringConverter implements AttributeConverter<String, String> {
	@Value("${db.cryp.key}")
	private String crypKey;
	@Value("${db.cryp.iv}")
	private String crypIv;
	
	private String encode = "UTF-8";

	@Override
	public String convertToDatabaseColumn(String attribute) {
		// TODO Auto-generated method stub
		if(attribute == null) return null;
		try {
			return encAES(attribute);
		}catch(Exception e) {
			return null;
		}
	}

	@Override
	public String convertToEntityAttribute(String dbData) {
		// TODO Auto-generated method stub
		if(dbData == null) return null;
		try {
			return decAES(dbData);
		}catch(Exception e) {
			return null;
		}
	}
	
	
	private Key getAESKey() throws Exception {
		Key keySpec;
		
		byte[] bytes = crypKey.getBytes(encode);
		
		keySpec = new SecretKeySpec(bytes, "AES");
		return keySpec;
	}
	
	// 암호화
	private String encAES(String str) throws Exception {
		Key keySpec = getAESKey();
		Cipher c = Cipher.getInstance("AES/CBC/PKCS5Padding");
		c.init(Cipher.ENCRYPT_MODE, keySpec, new IvParameterSpec(crypIv.getBytes(encode)));
		byte[] encryped = c.doFinal(str.getBytes(encode));
		String encStr = new String(Base64.getEncoder().encode(encryped));
		return encStr;
	}
	
	// 복호화
	public String decAES(String str) throws Exception {
		Key keySpec = getAESKey();
		Cipher c = Cipher.getInstance("AES/CBC/PKCS5Padding");
		c.init(Cipher.DECRYPT_MODE, keySpec, new IvParameterSpec(crypIv.getBytes(encode)));
		byte[] decryped = Base64.getDecoder().decode(str.getBytes(encode));
		String decStr = new String(c.doFinal(decryped), encode);
		return decStr;
	}
	

}
